<?php
declare(strict_types=1);

namespace App\Test\TestCase\Model\Table;

use App\Model\Table\ShelfCountsTable;
use Cake\TestSuite\TestCase;

/**
 * App\Model\Table\ShelfCountsTable Test Case
 */
class ShelfCountsTableTest extends TestCase
{
    /**
     * Test subject
     *
     * @var \App\Model\Table\ShelfCountsTable
     */
    protected $ShelfCounts;

    /**
     * Fixtures
     *
     * @var array<string>
     */
    protected array $fixtures = [
        'app.ShelfCounts',
    ];

    /**
     * setUp method
     *
     * @return void
     */
    protected function setUp(): void
    {
        parent::setUp();
        $config = $this->getTableLocator()->exists('ShelfCounts') ? [] : ['className' => ShelfCountsTable::class];
        $this->ShelfCounts = $this->getTableLocator()->get('ShelfCounts', $config);
    }

    /**
     * tearDown method
     *
     * @return void
     */
    protected function tearDown(): void
    {
        unset($this->ShelfCounts);

        parent::tearDown();
    }

    /**
     * Test validationDefault method
     *
     * @return void
     * @link \App\Model\Table\ShelfCountsTable::validationDefault()
     */
    public function testValidationDefault(): void
    {
        $this->markTestIncomplete('Not implemented yet.');
    }

    /**
     * Test buildRules method
     *
     * @return void
     * @link \App\Model\Table\ShelfCountsTable::buildRules()
     */
    public function testBuildRules(): void
    {
        $this->markTestIncomplete('Not implemented yet.');
    }
}
